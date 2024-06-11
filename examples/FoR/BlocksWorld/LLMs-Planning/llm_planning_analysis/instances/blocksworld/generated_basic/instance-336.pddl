

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(on a d)
(ontable b)
(ontable c)
(ontable d)
(clear a)
(clear b)
(clear c)
)
(:goal
(and
(on a c)
(on c d))
)
)


