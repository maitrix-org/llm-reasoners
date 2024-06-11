

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(on a c)
(ontable b)
(ontable c)
(on d a)
(clear b)
(clear d)
)
(:goal
(and
(on b c)
(on d a))
)
)


