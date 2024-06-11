

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(on a b)
(ontable b)
(ontable c)
(on d a)
(clear c)
(clear d)
)
(:goal
(and
(on a c)
(on d b))
)
)


