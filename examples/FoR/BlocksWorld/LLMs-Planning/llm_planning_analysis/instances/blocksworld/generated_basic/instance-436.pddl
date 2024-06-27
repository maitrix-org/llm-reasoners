

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(ontable a)
(ontable b)
(on c a)
(ontable d)
(clear b)
(clear c)
(clear d)
)
(:goal
(and
(on a b))
)
)


