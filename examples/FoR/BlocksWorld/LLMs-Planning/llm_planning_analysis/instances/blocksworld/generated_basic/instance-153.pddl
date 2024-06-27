

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(ontable a)
(ontable b)
(on c d)
(on d b)
(clear a)
(clear c)
)
(:goal
(and
(on a d)
(on b c))
)
)


