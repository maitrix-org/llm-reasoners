

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(on a c)
(on b a)
(ontable c)
(ontable d)
(clear b)
(clear d)
)
(:goal
(and
(on a b)
(on b d)
(on d c))
)
)


