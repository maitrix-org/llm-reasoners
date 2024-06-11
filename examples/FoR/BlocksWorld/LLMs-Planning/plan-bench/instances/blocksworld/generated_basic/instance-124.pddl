

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(on a b)
(ontable b)
(ontable c)
(ontable d)
(clear a)
(clear c)
(clear d)
)
(:goal
(and
(on c a))
)
)


