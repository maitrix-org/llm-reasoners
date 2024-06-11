

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(on a b)
(on b c)
(ontable c)
(on d a)
(clear d)
)
(:goal
(and
(on a c)
(on b d))
)
)


