

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(on a d)
(ontable b)
(ontable c)
(on d b)
(clear a)
(clear c)
)
(:goal
(and
(on a b)
(on c d))
)
)


